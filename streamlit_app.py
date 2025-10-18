with tab_racing:
    st.subheader("🏇 Racing Dudes Metrics")

    # --- Safely check for campaign_tags column ---
    if "campaign_tags" not in df.columns:
        st.warning("No `campaign_tags` column found in the dataset.")
    else:
        # Normalize the campaign_tags column to string for matching
        fdf_racing = df.copy()
        fdf_racing["campaign_tags"] = fdf_racing["campaign_tags"].astype(str).str.lower()

        # Filter for any users that contain 'racing_dudes'
        racing_df = fdf_racing[fdf_racing["campaign_tags"].str.contains("racing_dudes", na=False)].copy()

        # Further filter out junk/test emails
        if "email" in racing_df.columns:
            exclude_keywords = ["test", "yopmail", "ralls"]
            pattern = "|".join(exclude_keywords)
            racing_df = racing_df[~racing_df["email"].astype(str).str.contains(pattern, case=False, na=False)]

        if racing_df.empty:
            st.info("No valid users found with campaign_tags containing 'racing_dudes'.")
        else:
            st.caption(f"Found **{len(racing_df):,}** Racing Dudes user(s).")

            # --- Basic User Stats ---
            total_players = len(racing_df)
            ps_norm = racing_df["profile_status"].astype(str).str.strip().str.lower() if "profile_status" in racing_df.columns else pd.Series([], dtype="object")

            unverified_set = {"unverified"}
            kyc_set = {"grade-i", "grade-ii", "grade-iii"}

            unverified_count = int(ps_norm.isin(unverified_set).sum())
            kyc_verified_count = int(ps_norm.isin(kyc_set).sum())
            banned_count = int(ps_norm.str.contains("banned", na=False).sum())

            # Display simple player verification stats
            kpi_row = st.columns(4)
            kpi_row[0].metric("Players", f"{total_players:,}")
            kpi_row[1].metric("Unverified", f"{unverified_count:,}")
            kpi_row[2].metric("KYC Verified", f"{kyc_verified_count:,}")
            kpi_row[3].metric("Banned", f"{banned_count:,}")

            st.markdown("---")

            # --- Charts for this segment ---
            if "createdAt" in racing_df.columns:
                created_parsed = pd.to_datetime(racing_df["createdAt"], errors="coerce")
                monthly = (
                    pd.DataFrame({"month": created_parsed.dt.to_period("M").dt.to_timestamp()})
                    .dropna()
                    .groupby("month").size().reset_index(name="new_players")
                    .sort_values("month")
                )

                ch_month = alt.Chart(monthly).mark_bar().encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("new_players:Q", title="New players"),
                ).properties(height=300, title="New Racing Dudes Players by Month")

                now_naive = pd.Timestamp.utcnow().tz_localize(None)
                cutoff_date = (now_naive - pd.Timedelta(days=14)).normalize()
                recent_mask = created_parsed >= cutoff_date
                recent = created_parsed[recent_mask]

                if recent.notna().any():
                    recent_by_day = (
                        pd.DataFrame({"day": recent.dt.floor("D")})
                        .groupby("day").size().reset_index(name="new_players")
                        .sort_values("day")
                    )

                    ch_recent = (
                        alt.Chart(recent_by_day)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("day:T", title="Date"),
                            y=alt.Y("new_players:Q", title="New players"),
                        )
                        .properties(height=300, title="New Racing Dudes Players (Last 14 Days)")
                    )

                    ch_cols = st.columns(2)
                    ch_cols[0].altair_chart(ch_month, use_container_width=True)
                    ch_cols[1].altair_chart(ch_recent, use_container_width=True)

            st.markdown("---")

            # --- Table for these users ---
            st.markdown("### Racing Dudes Players")
            show_cols = [
                "username", "name", "email", "country", "profile_status",
                "createdAt", "contests_count_total", "usd_wallet_balance"
            ]
            show_cols = [c for c in show_cols if c in racing_df.columns]
            st.dataframe(racing_df[show_cols], use_container_width=True, hide_index=True)

            # CSV export
            csv = racing_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Racing Dudes CSV",
                csv,
                file_name="racing_dudes_players.csv",
                mime="text/csv"
            )
