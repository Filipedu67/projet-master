"use client"

import EstimateLink from './EstimateLink';

export function EstimateNav() {

    return (
        <div className='flex gap-8'>
            <EstimateLink href="/estimate/sell-price" label="Selling price" />
            <EstimateLink href="/estimate/rent" label="Rent" />
            <EstimateLink href="/estimate/roi" label="ROI" />
        </div>
    )
}